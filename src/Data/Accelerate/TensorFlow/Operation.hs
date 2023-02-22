{-# LANGUAGE BangPatterns      #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs             #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies      #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use record patterns" #-}
{-# LANGUAGE LambdaCase #-}

module Data.Accelerate.TensorFlow.Operation where

import Data.Array.Accelerate.AST.Operation
import Data.Array.Accelerate.AST.LeftHandSide
import Data.Array.Accelerate.Backend
import Data.Array.Accelerate.Type
import Data.Array.Accelerate.Representation.Array
import Data.Array.Accelerate.Representation.Type
import Data.Array.Accelerate.Trafo.Var
import Data.Array.Accelerate.AST.Idx
import Data.Array.Accelerate.Trafo.Operation.Substitution
import Data.Array.Accelerate.AST.Environment
import Data.Array.Accelerate.Representation.Ground
import Data.Array.Accelerate.Trafo.Desugar
import Data.Array.Accelerate.Trafo.Exp.Substitution
import Data.Type.Equality
import Data.Array.Accelerate.Lifetime
import Foreign
import Data.Array.Accelerate.AST.Kernel (NoKernelMetadata)
import Data.Text.Prettyprint.Doc
import Data.Array.Accelerate.Pretty.Exp
    ( prettyConst, primOperator )
import Data.Array.Accelerate.Pretty.Print (Operator(..))
import Data.Array.Accelerate.Pretty.Type (prettyScalarType)
import Data.Array.Accelerate.Representation.Shape
import Data.Array.Accelerate.Smart (typeR, undef)
import GHC.Conc (TVar(TVar))
import Data.Array.Accelerate.Pretty.Print (Adoc)

newtype BufferIdx benv a = BIdx (Idx benv (Buffer a)) -- a is enkele binding uit benv
-- | Environment with indexes pointing to buffers

instance Sink BufferIdx where
  weaken w (BIdx idx) = BIdx (weaken w idx)

-- benv is buffer env
-- env is scalar env
-- in env staat een int, en en benv kan ik op zoek naar een buffer van ints
type BufferEnv benv env = Env (BufferIdx benv) env

weakenEnv :: Sink f => (env1 :> env') -> Env (f env1) env2 -> Env (f env') env2
weakenEnv w = mapEnv (weaken w)

-- forall is alleen nodig als je @s wilt gebruiken in de method definition
distributeBIdx :: forall env s. TypeR s -> GroundVars env (Buffers s) -> Distribute (BufferIdx env) s
distributeBIdx TupRunit _ = ()
distributeBIdx (TupRsingle s) (TupRsingle (Var _ idx))
  | Refl <- reprIsSingle @ScalarType @s @(BufferIdx env) s
  , Refl <- reprIsSingle @ScalarType @s @Buffer s
  = BIdx idx
distributeBIdx (TupRpair l1 l2) (TupRpair r1 r2) = (distributeBIdx l1 r1, distributeBIdx l2 r2)
distributeBIdx _ _ = error "impossible"

data ScatterFun where
  ScatterFunAdd :: ScatterFun
  ScatterFunMin :: ScatterFun
  deriving Show

prettyScatterFun :: ScatterFun -> Adoc
prettyScatterFun ScatterFunAdd = "ScatterFunAdd"
prettyScatterFun ScatterFunMin = "ScatterFunMin"

data TensorOp op where
  TConstant :: ScalarType s -> s -> TensorOp (Out sh s -> ())
  TPrimFun :: PrimFun (a -> b) -> TensorOp (In sh a -> Out sh b -> ())
  
  TId :: TensorOp (In sh a -> Out sh a -> ())
  TWhere :: TensorOp (In sh a -> Out sh sh -> ()) -- how to include > 0?
  -- how to encompass TensorFlow shapes, indices? (5, 6) denotes a shape, but [5, 6] too.
  --TReshape :: ShapeR sh' -> TensorOp (In sh a -> Out sh' a -> ())
  TTensorScatter :: ScatterFun -> TensorOp (Mut sh' s -> In sh sh' -> In sh s -> ())
  TBooleanMask :: ScalarType s -> TensorOp (In DIM1 s -> In DIM1 PrimBool -> Out DIM1 s -> ())

instance PrettyOp TensorOp where
  prettyOp (TConstant s e) = vsep ["TConst", prettyConst (TupRsingle s) e]
  prettyOp (TPrimFun f) = vsep ["TBinOp", opName (primOperator f) ]
  prettyOp TId          = "TId"
  prettyOp (TTensorScatter f) = vsep ["TTensorScatter", viaShow f]
  prettyOp (TBooleanMask t) = vsep ["TBooleanMask"]

instance NFData' TensorOp where
  rnf' !_ = ()

instance DesugarAcc TensorOp where
  mkMap (ArgFun (Lam lhs (Body body))) (ArgArray _ (ArrayR _ t) _ gvb) aOut =
    mkMapF (push' Empty (lhs, distributeBIdx t gvb)) body aOut
  mkMap _ _ _ = error "impossible"

  mkGenerate f aOut@(ArgArray _ (ArrayR sh _) gv _)
    | sh' <- shapeType sh
    , DeclareVars lhs w k <- declareVars $ buffersR sh'
    = aletUnique lhs (desugarAlloc (ArrayR sh sh') (fromGrounds gv)) $ -- missing: filling with indices
      mkMap (weaken w f) (ArgArray In (ArrayR sh sh') (weakenVars w gv) (k weakenId)) (weaken w aOut)

  -- The result array is initialised with the given defaults and 
  -- any further values that are permuted into the result array 
  -- are added to the current value using the given combination function.

  -- The combination function is given the new value being permuted as its first argument, 
  -- and the current value of the array as its second.
  mkPermute -- hoe til ik dit uit dim1? En is het verder goed?
    (ArgFun comb)
    defaults@(ArgArray _ (ArrayR (ShapeRsnoc ShapeRz) _) gv' gvb')
    perm
    source@(ArgArray _ (ArrayR (ShapeRsnoc ShapeRz) t) gv gvb) -- reshape x compute (shapeSize)
    | sh <- ShapeRsnoc ShapeRz
    , maybeSh <- TupRpair (TupRsingle scalarTypeWord8) (TupRpair TupRunit (shapeType sh))
    , DeclareVars lhs w k  <- declareVars $ buffersR maybeSh
    , DeclareVars lhs' w' k' <- declareVars $ buffersR (shapeType sh)
    , DeclareVars lhs'' w'' k'' <- declareVars $ buffersR t
    = -- 1) allocate permute indices array n
      aletUnique lhs (desugarAlloc (ArrayR sh maybeSh) (fromGrounds gv)) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (mkGenerate
        (weaken w perm)
        (ArgArray Out (ArrayR sh maybeSh) (weakenVars w gv) (k weakenId))
      ) $
      aletUnique lhs' (desugarAlloc (ArrayR sh (shapeType sh)) (fromGrounds (weakenVars w gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit -- toIndex
      (booleanMask (shapeType sh)                                                    --(weakenVars w' $ isJust (TupRpair TupRunit (shapeType sh)) (k weakenId)))
       (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w' .> w) gv) (isJust (TupRpair TupRunit (shapeType sh)) (k w')))
       (fromJust (shapeType sh) (k w'))
       (k' weakenId)
      ) $
      aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
      Alet (LeftHandSideWildcard TupRunit) TupRunit
      (booleanMask t
       (ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (isJust (TupRpair TupRunit (shapeType sh)) (k (w'' .> w'))))
       (weakenVars (w'' .> w' .> w) gvb)
       (k'' weakenId)
      ) $
      Exec (TTensorScatter scatterFun) (
        ArgArray Mut (ArrayR dim1 t) (weakenVars (w'' .> w' .> w) gv') (weakenVars (w'' .> w' .> w) gvb') :>: 
        ArgArray In  (ArrayR dim1 (shapeType sh)) (weakenVars (w'' .> w' .> w) gv) (k' w'') :>:
        ArgArray In  (ArrayR dim1 t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId) :>: 
        ArgsNil)
        where scatterFun = case comb of
                Lam (LeftHandSideSingle _) (Lam (LeftHandSideSingle _) (Body (PrimApp (PrimAdd _) (Pair (Evar (Var _ ZeroIdx)) (Evar (Var _ (SuccIdx ZeroIdx))))))) -> ScatterFunAdd -- swap idx?
                _ -> error "complex combination for permute not supported" 
                -- how to pattern match on plus?
      
      -- aletUnique lhs'' (desugarAlloc (ArrayR sh t) (fromGrounds (weakenVars (w' .> w) gv))) $
      -- Alet (LeftHandSideWildcard TupRunit) TupRunit
      -- (Exec
      --   (TBooleanMask t)
      --   (ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb) :>:
      --     ArgArray In (ArrayR sh (TupRsingle scalarTypeWord8)) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w') $ extract1 (TupRpair TupRunit (shapeType sh)) (k weakenId)) :>:
      --     ArgArray Out (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId) :>:
      --     ArgsNil))
      -- (Exec
      --   (TTensorScatter ScatterFunAdd)
      --   (ArgArray Mut (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (weakenVars (w'' .> w' .> w) gvb) :>: 
      --    ArgArray In (ArrayR sh (shapeType sh)) (weakenVars (w'' .> w' .> w) gv) (weakenVars w'' $ k' weakenId) :>: 
      --    ArgArray In (ArrayR sh t) (weakenVars (w'' .> w' .> w) gv) (k'' weakenId) :>: ArgsNil)
      -- )

  mkPermute _ _ _ _ = undefined
    -- = -- 1) allocate perm indices array (maybe sh')
    --   aletUnique lhs (desugarAlloc (ArrayR sh maybeSh') (fromGrounds gv)) $
    --   Alet (LeftHandSideWildcard TupRunit) TupRunit
    --   (mkGenerate
    --     (weaken w perm)
    --     (ArgArray Out (ArrayR sh maybeSh') (weakenVars w gv) (k weakenId))
    --   ) $
    --   aletUnique lhs' (desugarAlloc (ArrayR dim1 (shapeType sh')) (shapeExpVars dim1 (weakenVars w gv))) $
    --   Alet (LeftHandSideWildcard TupRunit) TupRunit
    --   (Exec
    --     TBooleanMask
    --     (ArgArray In (ArrayR dim1 (shapeType sh')) _ ((\case
    --       TupRsingle _ -> error "impossible"
    --       TupRpair _ (TupRpair _ x) -> x
    --       _ -> error "impossible") (k weakenId)) :>: 
    --      ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) _ (f (k weakenId)) :>: 
    --      ArgArray Out (ArrayR dim1 (shapeType sh')) _ (k' weakenId) :>: 
    --      ArgsNil)
    --   )
    --   _

booleanMask :: TypeR a -> Arg env (In DIM1 Word8) -> GroundVars env (Buffers a) -> GroundVars env (Buffers a) -> PreOpenAcc TensorOp env ()
booleanMask TupRunit _ _ gvb = Return gvb
booleanMask t@(TupRsingle s) aOut@(ArgArray _ _ gv _) gvbIn gvbOut = Exec (TBooleanMask s) (ArgArray In (ArrayR dim1 t) gv gvbIn :>: aOut :>: ArgArray Out (ArrayR dim1 t) gv gvbOut :>: ArgsNil)
booleanMask (TupRpair t1 t2) aOut (TupRpair gvbIn1 gvbIn2) (TupRpair gvbOut1 gvbOut2) = Alet (LeftHandSideWildcard TupRunit) TupRunit 
 (booleanMask t1 aOut gvbIn1 gvbOut1)
 (booleanMask t2 aOut gvbIn2 gvbOut2)
booleanMask _ _ _ _ = error "impossible"

isJust :: TypeR a -> GroundVars env (Buffers (Word8, a)) -> GroundVars env (Buffers Word8)
isJust _ (TupRpair word8 _) = word8
isJust _ _ = error "impossible"

fromJust :: TypeR a -> GroundVars env (Buffers (Word8, ((), a))) -> GroundVars env (Buffers a)
fromJust _ (TupRpair _ (TupRpair _ a)) = a
fromJust _ _ = error "impossible"


      -- 3) flatten maybe indices, get boolean mask
      -- aletUnique lhs' (desugarAlloc (ArrayR dim1 (TupRsingle scalarTypeWord8)) _) $
      -- Alet (LeftHandSideWildcard TupRunit) TupRunit
      -- (mkMap 
      --   _ 
      --   (_ (ArgArray In (ArrayR sh maybeSh') _ (k weakenId)))
      --   (ArgArray Out (ArrayR sh (TupRsingle scalarTypeWord8)) _ (k' weakenId))
      -- )
      -- (Exec
      --   TBooleanMask
      --   (_ :>: 
      --    ArgArray In (ArrayR dim1 (TupRsingle scalarTypeWord8)) _ (k' weakenId) :>: 
      --    _ :>: 
      --    ArgsNil)
      -- )
      -- 4) apply boolean mask source and maybe indices
      -- 5) scatter 

      -- 2) zip with original values (src, maybe sh')
      -- aletUnique lhs' (desugarAlloc (ArrayR sh (TupRpair sht primMaybeSh')) (fromGrounds (weakenVars w gv))) $
      -- Alet (LeftHandSideWildcard TupRunit) TupRunit
      -- (mkZip 
      --   (weaken (w' .> w) source)
      --   (ArgArray In (ArrayR sh primMaybeSh') (weakenVars (w' .> w) gv) (k' weakenId)) 
      --   (ArgArray Out (ArrayR sh (TupRpair _ primMaybeSh')) (weakenVars (w' .> w) gv) (k' weakenId))
      -- )
      -- 3) map (src, nothing) -> (default, ?), 
          --    (src, just j) -> (comb(src, defaults[j]), j)
      -- (mkZip 
      --   (weaken w source) 
      --   (ArgArray In (ArrayR sh primMaybeSh') (weakenVars w gv) (k weakenId)) 
      --   _
      -- ) 
      -- (mkMap 
      --   _ 
      --   (ArgArray In (ArrayR sh primMaybeSh') (weakenVars w gv) (k weakenId)) 
      --   (ArgArray Out (ArrayR sh t) (weakenVars w gv) (k weakenId))
      -- )
      --Alet (LeftHandSideWildcard TupRunit) TupRunit
      --2) fill indices array
      -- (mkGenerate 
      --  (weaken w (ArgFun (Lam lhs' $ Body $ (case apply1 primMaybeSh' perm vars of
      --     Let lhs2 poe poe' -> error "1"
      --     _ -> error "2"
      --  )))) --(Alam lhs' $ Abody $ apply1 perm lhs')
      --  (ArgArray Out (ArrayR sh sh't) (weakenVars w gv) (k weakenId)))
      --_ 

      --_ --(mkBackpermute (Arg env (Fun' (sh' -> sh))) (Arg env (In sh t)) (Arg env (Out sh' t)) _ _ _ _)
      --_


  --   , sh't <- shapeType sh'
  --   , sh'mt < (Word8, ((), shapeType sh'))
  --   , DeclareVars lhs w k <- declareVars primConstType (PrimConst a) sh't
  --   , LHS shi _ <- mkLHS sh't
  --   = -- 1) allocate scatter indices array
  --     aletUnique lhs (desugarAlloc (ArrayR sh (primMaybify sh't)) (fromGrounds gv)) $
  --     Alet (LeftHandSideWildcard TupRunit) TupRunit
  --     (mkGenerate (weaken w perm) (ArgArray Out (ArrayR sh sh't) (weakenVars w gv) _))
  -- --       _
  --       -- 2) fill 
          -- mkGenerate (ArgFun $
          --   Lam lhs $ _
          -- ) (ArgArray Out arrayR gv' gvb')

        -- 1) allocate array with scatter indices
        --aletUnique lhs (desugarAlloc (ArrayR sh sh't) (fromGrounds gv)) $
        -- 2) fill output array with defaults
        --Alet (LeftHandSideWildcard TupRunit) TupRunit
        --  (mkGenerate (weaken w perm) (ArgArray Out (ArrayR sh sh't) _ _))
        --  _
        -- 3) allocate alternative source array
        -- 4) fill source array according to perm indices
        -- 5) 

          --     Alet (LeftHandSideWildcard TupRunit) TupRunit
          -- (mkGenerate (ArgFun f) (ArgArray Out arrayR gv gvb)) -- \sh -> case perm sh of
          -- _


mkMapF :: forall env env' sh t. BufferEnv env env' -> PreOpenExp (ArrayInstr env) env' t
  -> Arg env (Out sh t) -> OperationAcc TensorOp env ()
mkMapF _ (Const s e) aOut = Exec (TConstant s e) $ aOut :>: ArgsNil

mkMapF env (PrimApp f exp) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF -- flatten higher-order expression
     (weakenEnv w env)
     (weakenArrayInstr w exp)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   Exec -- apply method to the result
    (TPrimFun f)
    (
      ArgArray In (ArrayR sh a) (weakenVars w gv) (k weakenId) :>:
      weaken w aOut :>:
      ArgsNil
    )

mkMapF env (Let elhs exp1 exp2) aOut@(ArgArray _ (ArrayR sh _) gv _)
 | a <- expType exp1
 , DeclareVars lhs w k <- declareVars $ buffersR a
 = aletUnique lhs (desugarAlloc (ArrayR sh a) (fromGrounds gv)) $
   Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF
     (weakenEnv w env)
     (weakenArrayInstr w exp1)
     (ArgArray Out (ArrayR sh a) (weakenVars w gv) (k weakenId))
   ) $
   mkMapF
     (push' (weakenEnv w env) (elhs, distributeBIdx a (k weakenId)))
     (weakenArrayInstr w exp2)
     (weaken w aOut)

mkMapF env (Evar (Var s idx)) (ArgArray _ arrayR gv gvb@(TupRsingle (Var groundR _)))
  | Refl <- reprIsSingle @ScalarType @t @Buffer s
  = let (BIdx idx') = prj' idx env
        gvb'        = TupRsingle (Var groundR idx')
    in  Exec
          TId
          (
            ArgArray In arrayR gv gvb' :>:
            ArgArray Out arrayR gv gvb :>:
            ArgsNil
          )

mkMapF env (Pair exp1 exp2) (ArgArray _ (ArrayR sh (TupRpair t1 t2)) gv (TupRpair gvb1 gvb2))
 = Alet (LeftHandSideWildcard TupRunit) TupRunit
   (mkMapF env exp1 (ArgArray Out (ArrayR sh t1) gv gvb1))
   (mkMapF env exp2 (ArgArray Out (ArrayR sh t2) gv gvb2))

-- TODO

mkMapF _ (Foreign _ _ _ _) _ = undefined
mkMapF _ Nil _ = Return TupRunit
--mkMapF env Nil (ArgArray _ arrayR gv gvb) = Return gvb
-- TId (ArgArray In arrayR gv gvb :>: ArgArray Out arrayR gv gvb :>: ArgsNil)
--mkMapF _ Nil (ArgArray _ _ _ gvb) = Return gvb
mkMapF _ (VecPack _ _) _ = undefined
mkMapF _ (VecUnpack _ _) _ = undefined
mkMapF _ (IndexSlice _ _ _) _ = undefined
mkMapF _ (IndexFull _ _ _) _ = undefined
mkMapF _ (ToIndex _ _ _) _ = undefined -- hoe bewijs ik dat sh = sh?
mkMapF _ (FromIndex _ _ _) _ = undefined
mkMapF _ (Case _ _ _) _ = undefined
mkMapF _ (Cond _ _ _) _ = undefined
mkMapF _ (While _ _ _) _ = undefined
mkMapF _ (PrimConst _) _ = undefined
mkMapF _ (ArrayInstr _ _) _ = undefined
mkMapF _ (ShapeSize _ _) _ = undefined
mkMapF _ (Undef _) _ = undefined
mkMapF _ (Coerce _ _ _) _ = undefined
mkMapF _ _ _ = error "impossible"


-- temp kernel for testing purposes

data TensorFlowKernel env where
  TensorFlowKernel
    :: { kernelId       :: Int
       , kernelFunction :: !(Lifetime (FunPtr  env))
       }
    -> TensorFlowKernel env

instance NFData' TensorFlowKernel where
  rnf' (TensorFlowKernel !_ fn) = unsafeGetValue fn `seq` ()

newtype TensorFlowKernelMetadata f =
  TensorFlowKernelMetadata { kernelArgsSize :: Int }

instance IsKernel TensorFlowKernel where
  type KernelOperation TensorFlowKernel = TensorOp -- goed
  type KernelMetadata  TensorFlowKernel = NoKernelMetadata -- goed
  compileKernel = undefined

instance PrettyKernel TensorFlowKernelMetadata where
  prettyKernel = PrettyKernelBody True $ \_ kernel -> ""
