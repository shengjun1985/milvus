// Copyright (C) 2019-2020 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under the License.
package datacoord

import (
	"context"
	"errors"
	"sync"

	"github.com/milvus-io/milvus/internal/types"
)

type sessionManager interface {
	// try get session, without retry
	getSession(addr string) (types.DataNode, error)
	// try get session from manager with addr, if not exists, create one
	getOrCreateSession(addr string) (types.DataNode, error)
	releaseSession(addr string)
	release()
}

type clusterSessionManager struct {
	sync.RWMutex
	ctx               context.Context
	sessions          map[string]types.DataNode
	dataClientCreator dataNodeCreatorFunc
}

func newClusterSessionManager(ctx context.Context, dataClientCreator dataNodeCreatorFunc) *clusterSessionManager {
	return &clusterSessionManager{
		ctx:               ctx,
		sessions:          make(map[string]types.DataNode),
		dataClientCreator: dataClientCreator,
	}
}

// getSession with out creation if not found
func (m *clusterSessionManager) getSession(addr string) (types.DataNode, error) {
	m.RLock()
	defer m.RUnlock()
	cli, has := m.sessions[addr]
	if has {
		return cli, nil
	}
	return nil, errors.New("not found")
}

func (m *clusterSessionManager) createSession(addr string) (types.DataNode, error) {
	cli, err := m.dataClientCreator(m.ctx, addr)
	if err != nil {
		return nil, err
	}
	if err := cli.Init(); err != nil {
		return nil, err
	}
	if err := cli.Start(); err != nil {
		return nil, err
	}
	m.Lock()
	m.sessions[addr] = cli
	m.Unlock()
	return cli, nil
}

// entry function
func (m *clusterSessionManager) getOrCreateSession(addr string) (types.DataNode, error) {
	m.RLock()
	dn, has := m.sessions[addr]
	m.RUnlock()
	if has {
		return dn, nil
	}
	// does not need double check, addr has outer sync.Map
	dn, err := m.createSession(addr)
	return dn, err
}

// // lock acquired
// func (m *clusterSessionManager) hasSession(addr string) bool {
// 	_, ok := m.sessions[addr]
// 	return ok
// }

func (m *clusterSessionManager) releaseSession(addr string) {
	m.Lock()
	defer m.Unlock()
	cli, ok := m.sessions[addr]
	if !ok {
		return
	}
	_ = cli.Stop()
	delete(m.sessions, addr)
}

func (m *clusterSessionManager) release() {
	m.Lock()
	defer m.Unlock()
	for _, cli := range m.sessions {
		_ = cli.Stop()
	}
	m.sessions = map[string]types.DataNode{}
}
